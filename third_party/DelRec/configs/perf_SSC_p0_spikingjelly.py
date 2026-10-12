from spikingjelly.activation_based import surrogate

from src.spikingjelly_compat import ParametricLIFNodeCompat
from src.utils import Triangle


class Config:
    dataset = "SSC"
    datasets_path = "data/SSC"  # relative to the repository root (or pass --datasets-path)
    seed = 0
    seeds = [42, 52, 72, 92, 102]
    modes = ["clean", "hajr", "asar", "noise_aug"]
    # modes = ["clean"]

    epochs = 100
    batch_size = 128
    bias = False
    use_batch_norm = False
    results_dir = ""

    hidden_layers = [256, 256, 256]
    input_size = 700 // 5
    output_size = 35

    recurrent_dropout_rate = 0.3
    feedforward_dropout_rate = 0.1

    init_ff_weights = "default"
    init_dcls_weights = "default"
    no_delay_in_first_layer = False
    no_delay_in_last_layer = False

    neuron_module = ParametricLIFNodeCompat
    backend = "torch"

    tau = 2.0
    v_threshold = 1.0
    v_reset = None
    surrogate_function = Triangle.apply
    detach_reset = False
    decay_input = False
    step_mode = "m"
    store_v_seq = True

    reg_surrogate_type = "triangle"
    reg_surrogate_alpha = 1.0
    lambda_hajr = 0.1
    lambda_asar = 5e-4
    rampup_epochs = 1
    noise_aug_cv = 0.15
    eval_cvs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    eval_mc_runs = 10
    quant_bits = [32, 16, 8, 6, 4, 3, 2]

    use_sig_p = True
    delay_std_init = 12
    init_rec_delay = "half_normal"
    rec_delay_init_gain = 1.0
    sigma_init = 10.0
    sigma_decay = 0.95

    DCLSversion = "v1"
    kernel_count = 1
    max_feedforward_delay = 25
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay % 2 == 1 else max_feedforward_delay + 1
    left_padding = max_feedforward_delay - 1
    right_padding = 0
    init_pos_a = -max_feedforward_delay // 2
    init_pos_b = max_feedforward_delay // 2

    optim = "adam"
    scheduler_weights = "onecycle"
    scheduler_pos = "onecycle"
    lr_w = 1e-3
    lr_positions = 5e-2
    weight_decay = 1e-5

    time_window = 250
    n_bins = 5
