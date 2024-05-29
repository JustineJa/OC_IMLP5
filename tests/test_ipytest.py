import pytest
pytest.autoconfig()

def test_pred():
    question = "how to install python on linux ?"
    answer = pred_fn(question, pipe)
    list = ['python', 'linux']
    assert len(answer) == len(list) and sorted(answer) == sorted(list)

pytest.run('-vv')
