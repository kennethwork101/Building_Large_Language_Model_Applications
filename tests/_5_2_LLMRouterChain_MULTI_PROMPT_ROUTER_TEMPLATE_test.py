from kwwutils import clock, printit

from uvprog2025.Building_Large_Language_Model_Applications.src.building_large_language_model_applications._5_2_LLMRouterChain_MULTI_PROMPT_ROUTER_TEMPLATE import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    responses = main(**options)
    printit(f"{model}: responses", responses)
    # match_key = "snow white"
    # for response in responses:
    #     if response["input"] == "What was the first Disney movie?":
    #         assert match_key in response["text"].lower()
    #         break
    # else:
    #     assert False, f"_Error: {match_key} not found"
    # match_key = "snow white"
    # for response in responses:
    #     print("Checking response:", response)
    #     if response["input"] == "What was the first Disney movie?":
    #         if match_key in response["text"].lower():
    #             break
    # else:
    #     assert False, f"_Error: {match_key} not found"

    match_key = "snow white"
    for response in responses:
        print(f"Checking response: {response['input']}")
        if response["input"] == "What was the first Disney movie?":
            print(f"Found matching input. Checking text: {response['text'].lower()}")
            if match_key in response["text"].lower():
                print(f"Match found for '{match_key}' in text.")
                break
    else:
        assert False, f"_Error: {match_key} not found"
