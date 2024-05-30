import pytest
from entrypoint import main

def test_pred():
    question = "how to install python on linux ?"
    answer = main(question)
    answer_list = answer.split(", ")
    list = ['python', 'linux']
    assert len(answer_list) == len(list) 
    assert sorted(answer_list) == sorted(list)
