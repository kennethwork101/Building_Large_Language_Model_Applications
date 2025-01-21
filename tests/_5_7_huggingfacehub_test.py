import json
import os
from pathlib import Path

from kwwutils import clock, printit
from langchain.schema import AIMessage
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_6_3_create_react_agent_DuckDuckGoSearchRun import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "What were the first and second Disney movie?"
    response = main(**options)
    printit(f"1 {model}: response", response)
    output = response["output"].lower()
    assert response["input"] == options["question"]
    assert "pinocchio" in output
    assert "snow white" in output