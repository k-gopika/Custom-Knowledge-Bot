from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import cohere

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a strong secret key

co = cohere.Client('9s5CvhgNuvtxGHDWjKvDiMo2VNEPX8MaM16Ks3kD')  # Replace with your actual key

class Form(FlaskForm):
    text = StringField('Ask something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route("/", methods=["GET", "POST"])
def home():
    form = Form()
    response_text = ""

    if form.validate_on_submit():
        text = form.text.data
        
        # Use chat instead of generate
        response = co.chat(
            model='command-nightly',
            message=text,
        )
        
        response_text = response.text

    return render_template("home.html", form=form, response=response_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
