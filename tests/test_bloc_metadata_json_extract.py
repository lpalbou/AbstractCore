from abstractcore.core.bloc_metadata import _extract_json_object


def test_extract_json_object_repairs_truncated_open_quote_tail() -> None:
    raw = (
        '{"t":"x","d":"y","kw":["a","b"],"tp":["c"],"kind":"s:Report","mod":"text","lang":"en",'
        '"q":{"snr":0.1,"clar":0.2,"coh":0.3,"conc":0.4,"struct":0.5,"arg":0.6,"evid":0.7},"k":"v","'
    )
    data = _extract_json_object(raw)
    assert isinstance(data, dict)
    assert data.get("t") == "x"
    assert data.get("d") == "y"


def test_extract_json_object_repairs_trailing_comma_and_missing_closers() -> None:
    raw = '{"t":"x","d":"y","kw":["a","b",],"tp":["c"],"kind":"s:Report","mod":"text","lang":"en","q":{"snr":0.5,"clar":0.4,"coh":0.3,"conc":0.2,"struct":0.1,"arg":0.6,"evid":0.7},}'
    data = _extract_json_object(raw)
    assert isinstance(data, dict)
    assert data.get("t") == "x"
    assert data.get("kw") == ["a", "b"]

