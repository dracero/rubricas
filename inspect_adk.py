try:
    from google.labs.adk.core import Agent, Model
    print("Base Agent methods:", dir(Agent))
    # print("Base Model methods:", dir(Model))
except ImportError as e:
    print(f"Import Error: {e}")
