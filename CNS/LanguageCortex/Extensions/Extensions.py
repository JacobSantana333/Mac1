def get_extension(intent):


    extensionResponses = []
    extension = None
    entities  = False
    extension_responses = []
    extension_function = ""

    extension_check = intent["extension"]["function"] if intent["extension"]["function"] !="" else None

    if extension_check != None:
        extension_responses = intent["extension"]["responses"]
        extension_function = intent["extension"]["function"]

    return extension_function, extension_responses


def utilize_extension(extension_function, extension_responses,entities):
    class_parts = extension_function.split(".")
    class_folder = class_parts[0]
    class_subfolder = class_parts[1]
    class_name = class_parts[2]
    class_function = class_parts[3]
    module = __import__("CNS.LanguageCortex." + class_folder + "." + class_subfolder + "." + class_name, globals(),
                        locals(), [class_name])
    extension_class = getattr(module, class_name)()
    response = getattr(extension_class, class_function)(extension_responses,entities)
    return response
