def detect_task(query):

    query = query.lower()

    if any(

        word in query

        for word in [

            "quiz",
            "mcq",
            "question paper"

        ]
    ):

        return "quiz"

    elif any(

        word in query

        for word in [

            "summary",
            "summarize",
            "revision"

        ]
    ):

        return "summary"

    elif any(

        word in query

        for word in [

            "10 marks",
            "long answer",
            "detailed"

        ]
    ):

        return "long_answer"

    elif any(

        word in query

        for word in [

            "roadmap",
            "steps",
            "how to"

        ]
    ):

        return "roadmap"

    return "qa"