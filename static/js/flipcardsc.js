// DOM Elements
const game = document.getElementById('game');
const flipsCounter = document.getElementById('flips');
const matchesCounter = document.getElementById('matches');
const timerDisplay = document.getElementById('timer');
const restartButton = document.getElementById('restart');
const playAgainButton = document.getElementById('playAgain');
const winMessage = document.getElementById('winMessage');
const finalFlips = document.getElementById('finalFlips');
const finalTime = document.getElementById('finalTime');

// Game state
const symbols = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼'];
let cards = [];
let flippedCards = [];
let matchedPairs = 0;
let flips = 0;
let canFlip = true;
let startTime;
let timerInterval;

// Initialize game
function initGame() {
    // Reset stats
    flips = 0;
    matchedPairs = 0;
    flippedCards = [];
    flipsCounter.textContent = flips;
    matchesCounter.textContent = matchedPairs;
    canFlip = true;
    clearInterval(timerInterval);
    timerDisplay.textContent = '00:00';
    winMessage.classList.remove('show');
    
    // Create card pairs
    cards = [...symbols, ...symbols];
    
    // Shuffle cards
    shuffleCards();
    
    // Clear game board
    game.innerHTML = '';
    
    // Create card elements
    cards.forEach((symbol, index) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.index = index;
        card.dataset.symbol = symbol;
        
        const cardFront = document.createElement('div');
        cardFront.className = 'card-face card-front';
        cardFront.textContent = '?';
        
        const cardBack = document.createElement('div');
        cardBack.className = 'card-face card-back';
        cardBack.textContent = symbol;
        
        card.appendChild(cardFront);
        card.appendChild(cardBack);
        
        card.addEventListener('click', flipCard);
        game.appendChild(card);
    });
    
    // Start timer when first card is clicked
    const firstCardClick = () => {
        startTimer();
        game.removeEventListener('click', firstCardClick);
    };
    
    game.addEventListener('click', firstCardClick);
}

// Shuffle cards
function shuffleCards() {
    for (let i = cards.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [cards[i], cards[j]] = [cards[j], cards[i]];
    }
}

// Start timer
function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
}

// Update timer
function updateTimer() {
    const currentTime = Date.now();
    const elapsedTime = Math.floor((currentTime - startTime) / 1000);
    const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
    const seconds = (elapsedTime % 60).toString().padStart(2, '0');
    timerDisplay.textContent = `${minutes}:${seconds}`;
}

// Flip card
function flipCard() {
    if (!canFlip) return;
    if (this.classList.contains('flipped')) return;
    if (flippedCards.length >= 2) return;
    
    this.classList.add('flipped');
    flippedCards.push(this);
    
    flips++;
    flipsCounter.textContent = flips;
    
    if (flippedCards.length === 2) {
        canFlip = false;
        checkForMatch();
    }
}

// Check for match
function checkForMatch() {
    const [card1, card2] = flippedCards;
    
    if (card1.dataset.symbol === card2.dataset.symbol) {
        // Match found
        setTimeout(() => {
            card1.classList.add('matched');
            card2.classList.add('matched');
            
            // Remove event listeners
            card1.removeEventListener('click', flipCard);
            card2.removeEventListener('click', flipCard);
            
            matchedPairs++;
            matchesCounter.textContent = matchedPairs;
            flippedCards = [];
            canFlip = true;
            
            // Check for game completion
            if (matchedPairs === symbols.length) {
                clearInterval(timerInterval);
                setTimeout(showWinMessage, 1000);
            }
        }, 500);
    } else {
        // No match
        setTimeout(() => {
            card1.classList.remove('flipped');
            card2.classList.remove('flipped');
            flippedCards = [];
            canFlip = true;
        }, 1000);
    }
}

// Show win message
function showWinMessage() {
    finalFlips.textContent = flips;
    finalTime.textContent = timerDisplay.textContent;
    winMessage.classList.add('show');
}

// Event listeners
restartButton.addEventListener('click', initGame);
playAgainButton.addEventListener('click', initGame);

// Start game on load
document.addEventListener('DOMContentLoaded', initGame);