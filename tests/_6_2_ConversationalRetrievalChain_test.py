import json
import os
from pathlib import Path

from kwwutils import clock, printit
from langchain.schema import AIMessage
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._6_2_ConversationalRetrievalChain import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    filename = "italy_travel.pdf"
    dirpath = os.path.dirname(os.path.abspath(__file__))
    dirpath = Path(dirpath).parent
    filename = os.path.join(dirpath, filename)   
    options["filename"] = filename
    response = main(**options)
    printit(f"1 {model}: response", response)
    printit(f"2 {model}: response.keys()", sorted(response.keys()))
    question = "What can I visit in India?"
    assert sorted(response.keys()) == ['answer', 'chat_history', 'question'] 
    assert response["question"] == question
    assert "taj mahal" in response["answer"].lower()