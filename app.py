from flask import Flask, render_template, request
from interview_analysis import run_interview_analysis

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():

    # Run the OpenCV webcam interview analysis — this will open the webcam window!
    final_score, final_label = run_interview_analysis()

    # After closing webcam window, render thank you page with results
    return render_template('thankyou.html', score=round(final_score, 2), label=final_label)

if __name__ == '__main__':
    app.run(debug=True)
