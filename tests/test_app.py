from app import create_prompt

def test_create_prompt():
    result = create_prompt("test", True, False, "2D")
    assert result == "test, cartoon, 2d"