import json
import re

from kwwutils import clock, printit
from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_4_lcel_SimpleSequentialChain import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "Cats and Dogs"
    response = main(**options)
    printit(f"1 {model}: response", response)
    output = json.loads(response.json())
    printit(f"2 {model}: output", output)
    mpat = r"gat(a|o)"
    assert output["type"] == "ai"
    assert re.search(mpat, output["content"].lower())
