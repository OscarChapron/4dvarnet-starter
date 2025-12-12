import hydra
from omegaconf import OmegaConf

# ── enable ${eval:...} in YAML ──────────────────────────────────────────
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {}), use_cache=True)

@hydra.main(config_path='config', config_name='main', version_base='1.3')
def main(cfg):
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()

