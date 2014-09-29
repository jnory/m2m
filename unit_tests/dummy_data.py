
def generate_phrase_table():
    import StringIO

    lines = [
        "f1 ||| e1 ||| 0.1 0.2 0.3 0.4 ||| 1-1 ||| 1 |||",
        "f1 f2 ||| e1 ||| 0.2 0.3 0.4 0.5 ||| 1-1 ||| 1 |||",
        "f1 f2 ||| e1 e2 ||| 0.3 0.4 0.5 0.6 ||| 1-1 ||| 1 |||",
        "f1 f2 ||| e1 e2 e3 ||| 0.4 0.5 0.6 0.7 ||| 1-1 2-3 ||| 1 |||",
        "f2 ||| e1 e2 e3 ||| 0.5 0.6 0.7 0.8 ||| 1-1 ||| 1 |||",
        "f2 ||| e2 e3 ||| 0.6 0.7 0.8 0.9 ||| 1-1 ||| 1 |||",
        "f2 ||| e3 ||| 0.7 0.8 0.9 0.1 ||| 1-1 ||| 1 |||",
        "f3 ||| e1 ||| 0.8 0.9 0.1 0.2 ||| 1-1 ||| 1 |||"
    ]
    data = '\n'.join(lines)

    io = StringIO.StringIO()
    io.write(data)
    io.seek(0)
    return lines, io

def generate_moses_ini():
    import StringIO

    lines = [
        "# ignored",
        "[feature]",
        "A",
        "B name=X",
        "C name=Y path=Z",
        "D path=W",
        "", # ignored
        "[weight]",
        "X=0.2",
        "Y=0.5 0.6",
        "A0=0.2 0.3 0.5",
        "D0= 0.4 0.7 0.2",
        "[distortion-limit]",
        "6",
        "[stack]",
        "100",
        "[abc]",
        "def",
        "ghi"
    ]

    data = '\n'.join(lines)

    io = StringIO.StringIO()
    io.write(data)
    io.seek(0)
    return lines, io
