from flask import Flask, render_template, request, jsonify
from capemb import Embedder, Vocabulary, load

app = Flask(__name__)

# Load the real ecommerce dataset
problem = load("ecommerce")

# Build the real embedding model
embedder = Embedder(
    Vocabulary(
        list(problem.caps.values()),
        [problem.goal]
    )
)

# Encode all capabilities once
encoded_caps = {
    name: embedder.encode(cap)
    for name, cap in problem.caps.items()
}

# Encode states and goal
encoded_states = {
    name: embedder.encode(state)
    for name, state in problem.states.items()
}

encoded_goal = embedder.encode(problem.goal)


@app.route("/")
def home():
    return render_template(
        "index.html",
        capabilities=list(problem.caps.keys()),
        states=list(problem.states.keys())
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    state_name = data["state"]
    capability1_name = data["capability1"]
    capability2_name = data["capability2"]

    state = encoded_states[state_name]
    cap1 = encoded_caps[capability1_name]
    cap2 = encoded_caps[capability2_name]

    # Real compatibility calculation
    compatibility = embedder.compat(cap1, cap2)

    # Real goal relevance
    relevance_scores = embedder.relevance_all(
        encoded_goal,
        state,
        list(encoded_caps.values())
    )

    capability_names = list(encoded_caps.keys())

    relevance = float(
        relevance_scores[capability_names.index(capability1_name)]
    )

    # Real similarity
    similarity = embedder.similarity(
        cap1,
        cap2,
        view="function"
    )

    # Real applicability
    applicability = embedder.applicability(
        state,
        cap1
    )

    # Try composition
    composition = None

    try:
        composed = embedder.compose(
            [cap1, cap2],
            strict=False
        )

        composition = {
            "name": composed.name,
            "dimension": int(len(composed.vec))
        }

    except Exception as e:
        composition = {
            "error": str(e)
        }

    return jsonify({
        "capability1": capability1_name,
        "capability2": capability2_name,

        "compatibility": compatibility["score"],
        "composable": compatibility["composable"],

        "goal_relevance": relevance,

        "similarity": similarity,

        "applicability": applicability["score"],
        "applicable": applicability["applicable"],

        "composition": composition
    })


if __name__ == "__main__":
    app.run(debug=True)