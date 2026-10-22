r"""
Log messages of the solvers (:mod:`pyfe3d.solver.logger`)
=========================================================

.. currentmodule:: pyfe3d.solver.logger

The messages of the non-linear static analyses are printed to the standard
output, indented with one tab per level, and returned, such that the callers
may keep them. Every function has a ``silent`` flag that suppresses the
printing.

"""


def msg(text, level=0, silent=False):
    r"""Print a log message

    ``text`` is printed after ``level`` tabs unless ``silent`` is ``True``.

    Parameters
    ----------
    text : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab. Default ``0``.
    silent : bool, optional
        If ``True`` the message is not printed. Default ``False``.

    Returns
    -------
    text : str
        The message, without the indentation.

    """
    if not silent:
        print('\t'*level + text)
    return text


def warn(text, level=0, silent=False):
    r"""Print a warning message, starting with ``WARNING:``

    The prefix ``WARNING: `` is added to ``text``, and the result is printed
    after ``level`` tabs unless ``silent`` is ``True``. The :mod:`warnings`
    module is not used, such that nothing is raised or recorded.

    Parameters
    ----------
    text : str
        The message.
    level : int, optional
        Indentation level, each level adds a tab. Default ``0``.
    silent : bool, optional
        If ``True`` the message is not printed. Default ``False``.

    Returns
    -------
    text : str
        The message with its prefix, without the indentation.

    """
    return msg('WARNING: ' + text, level=level, silent=silent)
