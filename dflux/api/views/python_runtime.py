import epicbox


def exec_code(files):
    """
    This method will allows us execute the python code blocks or python files.
    """

    limits = {"cputime": 1, "memory": 64}
    final_result = {}
    # docker composed image
    # epicbox.configure(profiles=[epicbox.Profile("python", "soulpage/mlstack")])
    epicbox.configure(
        profiles=[epicbox.Profile("python", "docker-mlstack_soulpage-mlstat")]
    )

    # iterate over files to be executed
    for req_file in files:
        # execute file through sandbox
        result = epicbox.run(
            "python", f"python3 {req_file['name']}", files=files, limits=limits
        )
        # convert binary data to string
        result["stdout"] = result["stdout"].decode("utf-8")
        result["stderr"] = result["stderr"].decode("utf-8")

        # bind executed result to the filename
        final_result[req_file["name"]] = result
    return final_result
