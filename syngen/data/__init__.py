def __dir__():
    from os import walk
    from os.path import dirname, abspath, join, splitext

    data = []

    root, _, files = next(walk(dirname(abspath(__file__))))
    for name, ext in map(splitext, files):
        if ext.lower() in (".csv", ".yaml"):
            data.append(join(root, name + ext))

    return data
