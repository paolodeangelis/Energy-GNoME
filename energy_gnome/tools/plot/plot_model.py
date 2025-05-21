import matplotlib.pyplot as plt

fontsize = plt.rcParams["font.size"]
textsize = int(round(plt.rcParams["font.size"] * 1.4))


def visualize_layers(model):
    layer_dst = dict(zip(["sc", "lin1", "tp", "lin2"], ["gate", "tp", "lin2", "gate"]))
    try:
        layers = model.mp.layers
    except:  # noqa
        layers = model.layers

    num_layers = len(layers)
    num_ops = max(
        [
            len([k for k in list(layers[i].first._modules.keys()) if k not in ["fc", "alpha"]])
            for i in range(num_layers - 1)
        ]
    )

    fig, ax = plt.subplots(num_layers, num_ops, figsize=(14, 3.5 * num_layers))
    for i in range(num_layers - 1):
        ops = layers[i].first._modules.copy()
        ops.pop("fc", None)
        ops.pop("alpha", None)
        for j, (k, v) in enumerate(ops.items()):
            ax[i, j].set_title(k, fontsize=textsize)
            v.cpu().visualize(ax=ax[i, j])
            ax[i, j].text(
                0.7,
                -0.15,
                "--> to " + layer_dst[k],
                fontsize=textsize - 2,
                transform=ax[i, j].transAxes,
            )

    layer_dst = dict(zip(["sc", "lin1", "tp", "lin2"], ["output", "tp", "lin2", "output"]))
    ops = layers[-1]._modules.copy()
    ops.pop("fc", None)
    ops.pop("alpha", None)
    for j, (k, v) in enumerate(ops.items()):
        ax[-1, j].set_title(k, fontsize=textsize)
        v.cpu().visualize(ax=ax[-1, j])
        ax[-1, j].text(
            0.7,
            -0.15,
            "--> to " + layer_dst[k],
            fontsize=textsize - 2,
            transform=ax[-1, j].transAxes,
        )

    fig.subplots_adjust(wspace=0.3, hspace=0.5)
