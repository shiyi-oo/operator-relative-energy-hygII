"""Optional synchronous observation of real forward states; no retained tensors."""


def emit(trace, step, role, x, **metadata):
    if trace is not None:
        trace(step=step, role=role, x=x, **metadata)
