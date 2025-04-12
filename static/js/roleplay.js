const questions = [
    {
      situation: "You see an older person standing while you're sitting.",
      question: "What do you do?",
      options: {
        A: "Ignore and stay seated.",
        B: "Offer your seat to the older person.",
        C: "Wait for someone else to offer.",
        D: "Tell them to wait their turn."
      },
      correct: "B",
      feedback: {
        A: "Oops! That's not respectful. Try again. ",
        B: "Yay! That’s polite and kind. Great job! ",
        C: "Almost! Taking initiative is better. ",
        D: "That’s not very nice. Let's be kinder. "
      }
    },
    {
      situation: "A classmate drops their books in the hallway.",
      question: "How do you respond?",
      options: {
        A: "Laugh and keep walking.",
        B: "Help them pick up their books.",
        C: "Take a photo and share it.",
        D: "Stand and watch silently."
      },
      correct: "B",
      feedback: {
        A: "That's not kind. Let's support others. ",
        B: "Perfect! Helping shows kindness. ",
        C: "Oh no! That’s really unkind. ",
        D: "Not helpful. Try to be supportive. 🧠"
      }
    }
  ];
  
  let currentQuestion = 0;
  let score = 0;
  
  const quizBox = document.getElementById('quiz-box');
  const nextBtn = document.getElementById('next-btn');
  const scoreDisplay = document.getElementById('score');
  
  function loadQuestion() {
    const q = questions[currentQuestion];
    quizBox.innerHTML = `
      <p><strong>Scene:</strong> ${q.situation}</p>
      <p>${q.question}</p>
      ${Object.entries(q.options).map(([key, val]) =>
        `<div class="option" data-key="${key}">${key}. ${val}</div>`
      ).join('')}
      <div id="feedback" class="feedback"></div>
    `;
  
    document.querySelectorAll('.option').forEach(option => {
      option.addEventListener('click', () => {
        const choice = option.getAttribute('data-key');
        showFeedback(choice);
      });
    });
  
    nextBtn.classList.add('hidden');
  }
  
  function showFeedback(choice) {
    const q = questions[currentQuestion];
    const feedback = q.feedback[choice];
    document.getElementById('feedback').innerText = feedback;
    
    if (choice === q.correct) score++;
  
    document.querySelectorAll('.option').forEach(opt => opt.style.pointerEvents = 'none');
    nextBtn.classList.remove('hidden');
  }
  
  nextBtn.addEventListener('click', () => {
    currentQuestion++;
    if (currentQuestion < questions.length) {
      loadQuestion();
    } else {
      quizBox.innerHTML = '';
      scoreDisplay.classList.remove('hidden');
      scoreDisplay.innerHTML = `Quiz complete! You got <strong>${score}</strong> out of <strong>${questions.length}</strong> correct! Great job!`;
      nextBtn.classList.add('hidden');
    }
  });
  
  loadQuestion();
