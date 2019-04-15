from flask import Flask

import neuron

app = Flask(__name__)
app.register_blueprint(neuron.bp)
app.config.from_mapping(
    SECRET_KEY='dev'
)

if __name__ == '__main__':
    app.run()
