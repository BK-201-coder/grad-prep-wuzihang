from pathlib import Path

BASE = Path('/root/autodl-tmp/patchtst_repro')
CODE = BASE / 'code/PatchTST_supervised'
PYTHON = BASE / 'environment/venv/bin/python'
PAPER = {
    'ETTh2': {96: (0.274, 0.336), 192: (0.339, 0.379), 336: (0.331, 0.380), 720: (0.379, 0.422)},
    'ETTm2': {96: (0.165, 0.255), 192: (0.220, 0.292), 336: (0.278, 0.329), 720: (0.367, 0.385)},
}

def config(dataset, horizon):
    hourly = dataset == 'ETTh2'
    return dict(random_seed=2021, is_training=1,
        root_path=str(BASE / 'data'), data_path=dataset + '.csv',
        model_id=f'{dataset}_336_{horizon}', model='PatchTST', data=dataset,
        features='M', target='OT', freq='h', seq_len=336, label_len=48,
        pred_len=horizon, enc_in=7, dec_in=7, c_out=7, e_layers=3,
        n_heads=4 if hourly else 16, d_model=16 if hourly else 128,
        d_ff=128 if hourly else 256, dropout=0.3 if hourly else 0.2,
        fc_dropout=0.3 if hourly else 0.2, head_dropout=0,
        patch_len=16, stride=8, padding_patch='end', revin=1, affine=0,
        subtract_last=0, decomposition=0, kernel_size=25, individual=0,
        des='Exp', train_epochs=100, patience=100 if hourly else 20,
        lradj='type3' if hourly else 'TST', pct_start=0.3 if hourly else 0.4,
        itr=1, batch_size=128, learning_rate=0.0001, num_workers=4, embed='timeF')

def command(dataset, horizon):
    options = config(dataset, horizon)
    return [str(PYTHON), '-u', str(CODE / 'run_longExp.py')] + [part for k, v in options.items() for part in ('--' + k, str(v))]
