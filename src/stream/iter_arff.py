import typing

from scipy.io import arff

from river import base
from river.stream import utils


def iter_arff(
    filepath_or_buffer,
    target: str = None,
    compression: str = "infer",
    converters: dict = None,
    drop: typing.List[str] = [],
    sparse: bool = False,
) -> base.typing.Stream:
    """Iterates over rows from an ARFF file.

    Parameters
    ----------
    filepath_or_buffer
        Either a string indicating the location of a file, or a buffer object that has a
        `read` method.
    target
        Name of the target field.
    compression
        For on-the-fly decompression of on-disk data. If this is set to 'infer' and
        `filepath_or_buffer` is a path, then the decompression method is inferred for the
        following extensions: '.gz', '.zip'.
    """

    # If a file is not opened, then we open it
    buffer = filepath_or_buffer
    if not hasattr(buffer, "read"):
        buffer = utils.open_filepath(buffer, compression)

    try:
        _, attrs = arff._arffread.read_header(buffer)
    except ValueError as e:
        msg = f"Error while parsing header, error was: {e}"
        raise arff.ParseArffError(msg)

    names = [attr.name for attr in attrs]

    def boolint(x):
        return bool(int(x))

    def _conversor(attr):
        if attr.type_name == "numeric":
            return float
        if (
            attr.type_name == "nominal"
            and set(attr.range) == {"0", "1"}
            and attr.name not in target
        ):
            return int
        if attr.type_name == "nominal" and set(attr.range) == {"0", "1"}:
            return boolint
        return None

    types = [_conversor(attr) for attr in attrs]
    handle_empty = {
        float: 0.0,
        int: 0,
        boolint: False,
        None: None,
    }

    for r in buffer:
        if len(r) <= 1:
            continue
        if sparse:
            line = [entry.split(" ") for entry in r.rstrip()[1:-1].split(",")]
            vals = [handle_empty[typ] for typ in types]
            for i, val in line:
                vals[int(i)] = types[int(i)](val)
            x = {name: val for name, val in zip(names, vals)}
        else:
            vals = r.rstrip().split(",")
            x = {name: typ(val) if typ else val for name, typ, val in zip(names, types, vals)}

        for i in drop:
            del x[i]

        # Cast the values to the given types
        if converters is not None:
            for i, t in converters.items():
                x[i] = t(x[i])

        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)

        yield x, y

    # Close the file if we opened it
    if buffer is not filepath_or_buffer:
        buffer.close()
