import os


pkg_share = os.path.join(os.path.dirname(__file__), "models/")

print(pkg_share)
checkpoint_dir   = pkg_share + 'checkpoints'
best_model_dir   = pkg_share + 'best_model'
logs_dir         = pkg_share + 'logs'



for d in (checkpoint_dir, best_model_dir, logs_dir):
    os.makedirs(d, exist_ok=True)
