"""Trace-based fitness scorers: SoTL, SoTL-E, Early-Stop.

Every scorer maps a TrainingTrace to a scalar where HIGHER = BETTER (loss sums
are negated, following NASLib's convention), so all methods rank-compare
directly against validation accuracy.
"""

FITNESS_SCORERS = {}


def register_fitness(name):
    def decorator(cls):
        cls.name = name
        FITNESS_SCORERS[name] = cls
        return cls
    return decorator


@register_fitness('sotl')
class SoTL:
    """Sum of Training Losses (Ru et al., 2021): all minibatch losses so far."""
    needs_val_curve = False
    needs_final_val = False

    def score(self, trace):
        return -float(sum(trace.minibatch_losses))


@register_fitness('sotl_e')
class SoTLE:
    """SoTL-E: losses over the most recent full epoch (epoch = len(train_queue)
    minibatches). With a budget shorter than one epoch no complete window
    exists and the slice degenerates to all losses seen so far, i.e. SoTL.
    """
    needs_val_curve = False
    needs_final_val = False

    def score(self, trace):
        return -float(sum(trace.minibatch_losses[-trace.epoch_len:]))


@register_fitness('early_stop')
class EarlyStop:
    """Validation accuracy at the current budget (White et al.'s Early Stop (ACC))."""
    needs_val_curve = False
    needs_final_val = True

    def score(self, trace):
        return float(trace.final_val_acc)
