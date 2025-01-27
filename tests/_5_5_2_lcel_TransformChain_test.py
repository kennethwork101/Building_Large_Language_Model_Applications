import json
import os
from pathlib import Path

from kwwutils import clock, printit

from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_5_2_lcel_TransformChain import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    filename = "Cats&Dogs.txt"
    dirpath = os.path.dirname(os.path.abspath(__file__))
    dirpath = Path(dirpath).parent
    filename = os.path.join(dirpath, filename)
    options["filename"] = filename
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "Cats and Dogs"
    response = main(**options)
    jsonstr = response.json()
    outdict = json.loads(jsonstr)
    content = outdict["content"].lower()
    printit(f"1 {model}: response", response)
    printit(f"2 {model}: outdict", outdict)
    printit(f"3 {model}: outdict type", type(outdict))
    printit(f"4 {model}: content", content)
    printit(f"4 {model}: content type", type(content))
    assert outdict["type"] == "ai"
    assert "cat" in content
    assert "dog" in content
    assert "silvester" in content
