"""Count successful tuning trials and bound numerical-failure recovery."""

class NumericalTrainingError(ValueError):
    """A configuration produced nonfinite scores, parameters, or logits."""


def complete_search(study, objective, target=30, max_attempts=300):
    """Count successful trials only; tolerate divergence, not unrelated bugs."""
    from optuna.trial import TrialState
    attempts = 0
    while len(study.get_trials(states=(TrialState.COMPLETE,))) < target:
        if attempts >= max_attempts:
            raise RuntimeError(f'Search still incomplete after {max_attempts} new attempts')
        study.optimize(objective, n_trials=1, gc_after_trial=True,
                       catch=(NumericalTrainingError,))
        attempts += 1

