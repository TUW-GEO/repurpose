import os
import inspect
import functools
import warnings
from collections import deque

def deprecated(message: str = None):
    """
    Decorator for classes or functions to mark them as deprecated.
    If the decorator is applied without a specific message (`@deprecated()`),
    the default warning is shown when using the function/class. To specify
    a custom message use it like:
        @deprecated('Don't use this function anymore!').

    Parameters
    ----------
    message : str, optional (default: None)
        Custom message to show with the DeprecationWarning.
    """
    def decorator(src):
        default_msg = f"{src.__module__} " \
                      f"{'class' if inspect.isclass(src) else 'method'} " \
                      f"'{src.__module__}.{src.__name__}' " \
                      f"is deprecated and will be removed soon."

        @functools.wraps(src)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)

            warnings.warn(
                default_msg if message is None else message,
                category=DeprecationWarning,
                stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return src(*args, **kwargs)

        return new_func

    return decorator

def find_first_at_depth(root_dir, depth, reverse=False):
    """
    Finds and returns the first or last element at the specified depth in a directory tree.

    This function performs a breadth-first search (BFS) of the directory structure,
    starting from the given root directory, and returns the name of the first or last
    (sorted) file or directory found at the specified depth. The elements at each level
    are processed in lexicographical order by default, but can be reversed.

    Parameters:
    -----------
    root_dir : str
        The path to the root directory from which to start searching.
    depth : int
        The target depth to search for an element.
        - A depth of 0 refers to the `root_dir` itself.
        - A depth of 1 refers to the immediate subdirectories/files in `root_dir`.
        - A depth of 2 refers to the subdirectories/files within those subdirectories, and so on.
    reverse : bool, optional (default: False)
        If `False`, the function returns the first element at the specified depth (lexicographically).
        If `True`, it returns the last element at the specified depth (reverse lexicographically).

    Returns:
    --------
    str or None
        The name of the first (or last, if `reverse=True`) file or directory found at the specified depth,
        or `None` if no such element exists.

    Raises:
    -------
    ValueError:
        If the `root_dir` is not a valid directory.

    Notes:
    ------
    - If depth is 0, it will return the root directory itself if valid.
    - If files are encountered before reaching the target depth, they are ignored.
    """
    # Ensure the root directory exists
    if not os.path.isdir(root_dir):
        raise ValueError(f"{root_dir} is not a valid directory")

    # Initialize queue for BFS: elements are tuples (current_path, current_depth)
    queue = deque([(root_dir, 0)])

    while queue:
        current_path, current_depth = queue.popleft()

        # If we have reached the target depth, return the first/last sorted element at this level
        if current_depth == depth:
            try:
                # Get the sorted list of files/directories
                elements = sorted(os.listdir(current_path), reverse=reverse)
                return elements[0]
            except IndexError:
                return None  # No elements at this level
            except NotADirectoryError:
                continue  # Skip files if they are encountered before depth

        # If not at target depth, enqueue the next level
        try:
            entries = sorted(os.listdir(current_path), reverse=reverse)
            for entry in entries:
                full_path = os.path.join(current_path, entry)
                if os.path.isdir(full_path):
                    queue.append((full_path, current_depth + 1))
        except NotADirectoryError:
            continue  # Skip if we encounter files before reaching depth

    return None  # If we exhaust all options and don't find anything


def delete_empty_directories(path: str):
    """
    Delete empty dirs in path
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the directory is empty
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
