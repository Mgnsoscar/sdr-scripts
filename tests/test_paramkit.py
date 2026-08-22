

def test_integer_accepts_leading_zero():
    from paramkit import Script
    s = Script("t").integer("-N", "--n", min=0, max=100)
    assert s.parse(["--n", "08"]).n == 8        # base-0 rejects '08'; we accept it
    assert s.parse(["--n", "0x10"]).n == 16     # hex still works
