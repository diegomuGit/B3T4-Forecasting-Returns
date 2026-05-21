"""
model_builder.py — Auto-detect and build models from .h5 checkpoints.

Reads the .h5 file to extract layer SIZES, then builds the model in the
correct architectural order (not h5 alphabetical order).

Supported architectures: CNN (conv1d), MLP, RNN (LSTM/GRU), Mixto.
"""

import os, re
import numpy as np


def _read_h5_layers(path: str) -> dict:
    """
    Read all layer names, types, and weight shapes from an .h5 file.
    Returns dict: {layer_name: [(shape, dtype), ...], ...}
    """
    import h5py
    layers = {}
    with h5py.File(path, 'r') as f:
        if 'layers' not in f:
            return layers
        for name in f['layers']:
            layer = f['layers'][name]
            weights = []
            # Direct vars
            if 'vars' in layer:
                for k in sorted(layer['vars'].keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    weights.append(layer['vars'][k].shape)
            # Nested (LSTM cell)
            if 'cell' in layer and 'vars' in layer['cell']:
                for k in sorted(layer['cell']['vars'].keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    weights.append(layer['cell']['vars'][k].shape)
            layers[name] = weights
    return layers


def _extract_architecture(h5_layers: dict) -> dict:
    """
    From h5 layer names and weight shapes, extract the architectural parameters.
    Returns a structured dict describing the architecture.
    """
    # Classify each layer
    convs = []    # (name, kernel_size, in_ch, filters)
    bns = []      # (name, features)
    denses = []   # (name, in_features, out_features)
    lstms = []    # (name, input_dim, units)
    grus = []     # (name, input_dim, units)
    has_gap = False
    has_flatten = False

    for name, weights in h5_layers.items():
        nl = name.lower()

        if ('conv1d' in nl or 'conv' in nl) and not ('batch' in nl):
            if len(weights) >= 2:
                ks, in_ch, filt = weights[0]
                convs.append((name, ks, in_ch, filt))

        elif 'batch_normalization' in nl or 'batch_norm' in nl or ('bn' in nl and 'conv' not in nl):
            if weights:
                bns.append((name, weights[0][0]))

        elif 'dense' in nl:
            if len(weights) >= 2:
                in_f, out_f = weights[0]
                denses.append((name, in_f, out_f))

        elif 'lstm' in nl:
            if len(weights) >= 3:
                in_dim = weights[0][0]
                units = weights[0][1] // 4
                lstms.append((name, in_dim, units))

        elif 'gru' in nl:
            if len(weights) >= 3:
                in_dim = weights[0][0]
                units = weights[0][1] // 3
                grus.append((name, in_dim, units))

        elif 'global_average_pooling' in nl:
            has_gap = True

        elif 'flatten' in nl:
            has_flatten = True

    # Sort convs by input channels (data flow order)
    convs.sort(key=lambda x: x[2])

    # Sort denses by tracing the data flow:
    # build a chain starting from the largest in_features
    if denses:
        dense_chain = _chain_denses(denses)
    else:
        dense_chain = []

    # Determine architecture type
    has_concat = any('concat' in name.lower() for name in h5_layers)
    has_conv = len(convs) > 0
    has_rnn = len(lstms) > 0 or len(grus) > 0

    if has_concat and has_conv and has_rnn:
        arch_type = 'mixto'
    elif has_conv:
        arch_type = 'cnn'
    elif has_rnn:
        arch_type = 'rnn'
    else:
        arch_type = 'mlp'

    return {
        'arch_type': arch_type,
        'convs': convs,
        'bns': bns,
        'denses': dense_chain,
        'lstms': lstms,
        'grus': grus,
        'has_gap': has_gap,
        'has_flatten': has_flatten,
    }


def _chain_denses(denses: list) -> list:
    """
    Order dense layers by data flow: trace in_features -> out_features chain.
    Returns sorted list of (name, in_f, out_f).
    """
    # Build lookup: in_features -> (name, in_f, out_f)
    remaining = list(denses)
    chain = []

    # Find the first dense: its in_features should NOT match any other dense's out_features
    out_set = {d[2] for d in remaining}
    starts = [d for d in remaining if d[1] not in out_set]

    if starts:
        current = starts[0]
    else:
        # Fallback: sort by in_features descending
        remaining.sort(key=lambda x: x[1], reverse=True)
        current = remaining[0]

    chain.append(current)
    remaining.remove(current)

    # Follow the chain
    while remaining:
        next_in = current[2]  # out_features of current = in_features of next
        found = [d for d in remaining if d[1] == next_in]
        if found:
            current = found[0]
            chain.append(current)
            remaining.remove(current)
        else:
            # No match — append remaining sorted by in_features descending
            remaining.sort(key=lambda x: x[1], reverse=True)
            chain.extend(remaining)
            break

    return chain


# ─────────────────────────────────────────────────────────────────────
# BUILD FUNCTIONS — correct architectural order
# ─────────────────────────────────────────────────────────────────────

def _build_cnn(arch_info: dict, v_in: int, n_assets: int):
    """Build CNN: Conv→BN→Drop blocks → GAP → Dense head."""
    from tensorflow import keras
    from tensorflow.keras import layers as L

    model_layers = [L.Input(shape=(v_in, n_assets))]

    # Conv + BN blocks (paired by filter count)
    bn_by_features = {feat: name for name, feat in arch_info['bns']}

    for name, ks, in_ch, filt in arch_info['convs']:
        model_layers.append(L.Conv1D(filt, ks, padding='same', activation='relu'))
        if filt in bn_by_features:
            model_layers.append(L.BatchNormalization())
        model_layers.append(L.Dropout(0.25))

    model_layers.append(L.GlobalAveragePooling1D())

    # Dense head
    for name, in_f, out_f in arch_info['denses']:
        is_output = (out_f == n_assets)
        model_layers.append(L.Dense(out_f, activation=None if is_output else 'relu'))
        if not is_output:
            model_layers.append(L.Dropout(0.25))

    return keras.Sequential(model_layers, name='cnn')


def _build_mlp(arch_info: dict, v_in: int, n_assets: int):
    """Build MLP: Flatten → Dense chain."""
    from tensorflow import keras
    from tensorflow.keras import layers as L

    model_layers = [L.Input(shape=(v_in, n_assets)), L.Flatten()]

    for name, in_f, out_f in arch_info['denses']:
        is_output = (out_f == n_assets)
        model_layers.append(L.Dense(out_f, activation=None if is_output else 'relu'))
        if not is_output:
            model_layers.append(L.Dropout(0.25))

    return keras.Sequential(model_layers, name='mlp')


def _build_rnn(arch_info: dict, v_in: int, n_assets: int):
    """Build RNN: LSTM/GRU → Dense head."""
    from tensorflow import keras
    from tensorflow.keras import layers as L

    model_layers = [L.Input(shape=(v_in, n_assets))]

    if arch_info['lstms']:
        _, _, units = arch_info['lstms'][0]
        model_layers.append(L.LSTM(units))
    elif arch_info['grus']:
        _, _, units = arch_info['grus'][0]
        model_layers.append(L.GRU(units))

    for name, in_f, out_f in arch_info['denses']:
        is_output = (out_f == n_assets)
        model_layers.append(L.Dense(out_f, activation=None if is_output else 'relu'))
        if not is_output:
            model_layers.append(L.Dropout(0.25))

    return keras.Sequential(model_layers, name='rnn')


def _build_mixto(arch_info: dict, v_in: int, n_assets: int):
    """Build Mixto: CNN + RNN + MLP branches → concat → Dense head."""
    from tensorflow import keras
    from tensorflow.keras import layers as L, regularizers

    l2 = regularizers.l2(1e-4)
    inputs = L.Input(shape=(v_in, n_assets), name='input_window')

    # CNN branch
    conv_filters = arch_info['convs'][0][3] if arch_info['convs'] else 64
    c = L.Conv1D(conv_filters, 3, padding='same', activation='relu',
                 kernel_regularizer=l2, name='cnn_conv1d')(inputs)
    c = L.BatchNormalization(name='cnn_bn')(c)
    c = L.Dropout(0.25, name='cnn_dropout')(c)
    c = L.GlobalAveragePooling1D(name='cnn_gap')(c)

    # RNN branch
    rnn_units = arch_info['lstms'][0][2] if arch_info['lstms'] else 64
    if arch_info['grus']:
        r = L.GRU(rnn_units, name='rnn_gru')(inputs)
    else:
        r = L.LSTM(rnn_units, recurrent_dropout=0.10, name='rnn_lstm')(inputs)

    # MLP branch — find the dense with largest in_features (post-flatten)
    mlp_denses = [d for d in arch_info['denses'] if d[1] > n_assets * 2 and d[2] != n_assets]
    mlp_units = mlp_denses[0][2] if mlp_denses else 128
    m = L.Flatten(name='mlp_flatten')(inputs)
    m = L.Dense(mlp_units, activation='relu', kernel_regularizer=l2, name='mlp_dense')(m)
    m = L.Dropout(0.25, name='mlp_dropout')(m)

    # Head
    head_denses = [d for d in arch_info['denses'] if d not in mlp_denses and d[2] != n_assets]
    head_units = head_denses[0][2] if head_denses else 64
    x = L.Concatenate(name='fusion_concat')([c, r, m])
    x = L.Dense(head_units, activation='relu', kernel_regularizer=l2, name='head_dense')(x)
    x = L.Dropout(0.35, name='head_dropout')(x)
    outputs = L.Dense(n_assets, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mixto')
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mae', metrics=['mae'])
    return model


# ─────────────────────────────────────────────────────────────────────
# MAIN API
# ─────────────────────────────────────────────────────────────────────

_BUILDERS = {
    'cnn': _build_cnn,
    'mlp': _build_mlp,
    'rnn': _build_rnn,
    'mixto': _build_mixto,
}


def build_and_load_from_h5(path: str, v_in: int, n_assets: int = 23):
    """
    Auto-detect architecture from .h5, build matching model, load weights.

    IMPORTANT: Clears the Keras session before building to avoid layer
    name collisions when called multiple times.

    Returns (model, arch_type).
    """
    import tensorflow as tf

    # Clear session to reset layer name counters
    tf.keras.backend.clear_session()

    # Read and classify
    h5_layers = _read_h5_layers(path)
    arch_info = _extract_architecture(h5_layers)
    arch_type = arch_info['arch_type']

    # Build
    builder = _BUILDERS.get(arch_type)
    if builder is None:
        raise ValueError(f'Unknown architecture type: {arch_type}')
    model = builder(arch_info, v_in, n_assets)
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Load weights
    model.load_weights(path)

    print(f'> Auto-built {arch_type}: {model.count_params():,} params from {os.path.basename(path)}')
    return model, arch_type


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────────────

def discover_checkpoints(base: str, v_out_filter: int = 90):
    """Scan 08_results/checkpoints/ for inv_* checkpoints."""
    ckpt_dir = os.path.join(base, '08_results', 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return []

    pattern = re.compile(r'^inv_(\w+)_vin(\d+)_vout(\d+)_best\.weights\.h5$')
    results = []
    for f in sorted(os.listdir(ckpt_dir)):
        m = pattern.match(f)
        if m:
            arch_raw, v_in, v_out = m.group(1), int(m.group(2)), int(m.group(3))
            if v_out_filter and v_out != v_out_filter:
                continue
            results.append({
                'arch_raw': arch_raw, 'v_in': v_in, 'v_out': v_out,
                'modelo': f'inv_{arch_raw}',
                'run_tag': f'inv_{arch_raw}_vin{v_in}_vout{v_out}',
                'filename': f,
                'path': os.path.join(ckpt_dir, f),
            })
    return results


# ─────────────────────────────────────────────────────────────────────
# LEGACY API (backward compat with 04_full_pipeline)
# ─────────────────────────────────────────────────────────────────────

def build_model(arch: str, v_in: int, n_assets: int = 23):
    """Legacy: build model by architecture name (fixed architecture)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers as L

    tf.keras.backend.clear_session()

    if arch in ('cnn', 'conv1d'):
        return keras.Sequential([
            L.Input(shape=(v_in, n_assets)),
            L.Conv1D(64, 3, padding='same', activation='relu'),
            L.BatchNormalization(), L.Dropout(0.25),
            L.Conv1D(128, 3, padding='same', activation='relu'),
            L.BatchNormalization(), L.Dropout(0.25),
            L.GlobalAveragePooling1D(),
            L.Dense(64, activation='relu'), L.Dropout(0.25),
            L.Dense(n_assets),
        ], name='cnn')
    elif arch == 'mlp':
        return keras.Sequential([
            L.Input(shape=(v_in, n_assets)), L.Flatten(),
            L.Dense(256, activation='relu'), L.Dropout(0.25),
            L.Dense(128, activation='relu'), L.Dropout(0.25),
            L.Dense(n_assets),
        ], name='mlp')
    elif arch == 'rnn':
        return keras.Sequential([
            L.Input(shape=(v_in, n_assets)),
            L.LSTM(64),
            L.Dense(64, activation='relu'), L.Dropout(0.25),
            L.Dense(n_assets),
        ], name='rnn')
    else:
        raise ValueError(f'Unknown arch: {arch}')
