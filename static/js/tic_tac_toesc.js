document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const cells = document.querySelectorAll('.cell');
    const board = document.getElementById('board');
    const status = document.getElementById('status');
    const resetButton = document.getElementById('reset-btn');
    const restartButton = document.getElementById('restart-btn');
    const xScoreDisplay = document.getElementById('x-score');
    const oScoreDisplay = document.getElementById('o-score');
    const currentLevelDisplay = document.getElementById('current-level');
    const levelNameDisplay = document.getElementById('level-name');
    const totalWinsDisplay = document.getElementById('total-wins');
    const levelUpModal = document.getElementById('level-up-modal');
    const newLevelName = document.getElementById('new-level-name');
    const continueButton = document.getElementById('continue-btn');
    const overlay = document.getElementById('overlay');
    const levelProgress = document.getElementById('level-progress');
    const stars = [
        document.getElementById('star1'),
        document.getElementById('star2'),
        document.getElementById('star3')
    ];
    
    // Game variables
    let currentPlayer = 'X';
    let gameBoard = Array(9).fill('');
    let gameActive = true;
    let scores = { X: 0, O: 0 };
    let playerSymbol = 'X';
    let aiSymbol = 'O';
    let currentLevel = 1;
    let totalWins = 0;
    
    // Level configuration
    const levels = {
        1: { name: "Rookie", winsToNextLevel: 3, difficulty: "easy" },
        2: { name: "Explorer", winsToNextLevel: 5, difficulty: "medium" },
        3: { name: "Champion", winsToNextLevel: null, difficulty: "hard" }
    };
    
    // Define winning combinations
    const winCombos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
        [0, 4, 8], [2, 4, 6]             // diagonals
    ];
    
    // Initialize the game
    function initGame() {
        gameBoard = Array(9).fill('');
        gameActive = true;
        currentPlayer = playerSymbol;
        updateStatusMessage();
        
        cells.forEach(cell => {
            cell.className = 'cell';
            cell.textContent = '';
        });
        
        updateLevelInfo();
    }
    
    // Update level information display
    function updateLevelInfo() {
        currentLevelDisplay.textContent = currentLevel;
        levelNameDisplay.textContent = levels[currentLevel].name;
        
        // Update stars based on level
        stars.forEach((star, index) => {
            star.classList.toggle('filled', index < currentLevel);
        });
        
        // Calculate and update progress bar
        if (levels[currentLevel].winsToNextLevel !== null) {
            const levelWins = totalWins % levels[currentLevel].winsToNextLevel || 0;
            const progressPercentage = (levelWins / levels[currentLevel].winsToNextLevel) * 100;
            levelProgress.style.width = `${progressPercentage}%`;
        } else {
            levelProgress.style.width = '100%';
        }
        
        totalWinsDisplay.textContent = totalWins;
    }
    
    // Update status message
    function updateStatusMessage() {
        if (currentPlayer === playerSymbol) {
            status.textContent = "Your turn!";
        } else {
            const messages = ["Robot is thinking...", "Hmm...", "Robot's turn"];
            status.textContent = messages[Math.floor(Math.random() * messages.length)];
        }
    }
    
    // Create confetti effect
    function createConfetti() {
        const confettiCount = 100;
        const colors = ['#FF6B6B', '#4ECDC4', '#FFD166', '#6A67CE', '#FF9FB1'];
        
        for (let i = 0; i < confettiCount; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.left = `${Math.random() * 100}vw`;
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.opacity = '1';
            confetti.style.animation = `confetti-fall ${Math.random() * 3 + 2}s linear forwards`;
            document.body.appendChild(confetti);
            
            // Remove confetti after animation completes
            setTimeout(() => {
                document.body.removeChild(confetti);
            }, 5000);
        }
    }
    
    // Handle player move
    function handleCellClick(e) {
        const index = e.target.dataset.index;
        
        if (gameBoard[index] !== '' || !gameActive || currentPlayer !== playerSymbol) {
            return;
        }
        
        makeMove(index);
        
        // If game is still active and it's AI's turn
        if (gameActive && currentPlayer === aiSymbol) {
            // Add a short delay for AI move to make it seem like it's "thinking"
            board.classList.add('ai-thinking');
            setTimeout(() => {
                makeAIMove();
                board.classList.remove('ai-thinking');
            }, 800);
        }
    }
    
    // Make a move
    function makeMove(index) {
        gameBoard[index] = currentPlayer;
        cells[index].classList.add(currentPlayer.toLowerCase());
        
        // Check for win or draw
        if (checkWin()) {
            if (currentPlayer === playerSymbol) {
                status.textContent = "You win! Awesome!";
                createConfetti();
                
                // Handle player win
                totalWins++;
                scores[currentPlayer]++;
                updateScores();
                
                // Check for level up
                const currentLevelConfig = levels[currentLevel];
                if (currentLevelConfig.winsToNextLevel !== null && 
                    totalWins % currentLevelConfig.winsToNextLevel === 0) {
                    if (currentLevel < Object.keys(levels).length) {
                        setTimeout(levelUp, 1500);
                    }
                } else {
                    updateLevelInfo();
                }
            } else {
                status.textContent = "Robot wins! Try again!";
                scores[currentPlayer]++;
                updateScores();
            }
            
            gameActive = false;
            return;
        }
        
        if (checkDraw()) {
            status.textContent = "It's a tie! Nice try!";
            gameActive = false;
            return;
        }
        
        // Switch player
        currentPlayer = currentPlayer === playerSymbol ? aiSymbol : playerSymbol;
        updateStatusMessage();
    }
    
    // Level up function
    function levelUp() {
        currentLevel++;
        overlay.style.display = 'block';
        newLevelName.textContent = levels[currentLevel].name;
        levelUpModal.classList.add('show');
        createConfetti();
    }
    
    // Continue after level up
    function continueLevelUp() {
        overlay.style.display = 'none';
        levelUpModal.classList.remove('show');
        updateLevelInfo();
        initGame();
    }
    
    // AI move logic
    function makeAIMove() {
        if (!gameActive) return;
        
        let index;
        const difficulty = levels[currentLevel].difficulty;
        
        switch (difficulty) {
            case 'easy':
                index = makeRandomMove();
                break;
            case 'medium':
                // 60% chance to make smart move, 40% chance to make random move
                index = Math.random() < 0.6 ? makeSmartMove() : makeRandomMove();
                break;
            case 'hard':
                index = makeSmartMove();
                break;
            default:
                index = makeRandomMove();
        }
        
        makeMove(index);
    }
    
    // Make a random move (for easy AI)
    function makeRandomMove() {
        const emptyIndices = gameBoard
            .map((cell, index) => cell === '' ? index : null)
            .filter(index => index !== null);
        
        return emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
    }
    
    // Make a smart move (for medium and hard AI)
    function makeSmartMove() {
        // Check if AI can win in one move
        for (let i = 0; i < gameBoard.length; i++) {
            if (gameBoard[i] === '') {
                gameBoard[i] = aiSymbol;
                if (checkForWin(aiSymbol)) {
                    gameBoard[i] = '';
                    return i;
                }
                gameBoard[i] = '';
            }
        }
        
        // Check if player can win in one move and block
        for (let i = 0; i < gameBoard.length; i++) {
            if (gameBoard[i] === '') {
                gameBoard[i] = playerSymbol;
                if (checkForWin(playerSymbol)) {
                    gameBoard[i] = '';
                    return i;
                }
                gameBoard[i] = '';
            }
        }
        
        // Try to take center
        if (gameBoard[4] === '') {
            return 4;
        }
        
        // Try to take corners
        const corners = [0, 2, 6, 8];
        const availableCorners = corners.filter(corner => gameBoard[corner] === '');
        if (availableCorners.length > 0) {
            return availableCorners[Math.floor(Math.random() * availableCorners.length)];
        }
        
        // Take sides
        const sides = [1, 3, 5, 7];
        const availableSides = sides.filter(side => gameBoard[side] === '');
        if (availableSides.length > 0) {
            return availableSides[Math.floor(Math.random() * availableSides.length)];
        }
        
        // If all else fails, make a random move
        return makeRandomMove();
    }
    
    // Check if a player has won
    function checkForWin(player) {
        return winCombos.some(combo => {
            return combo.every(index => gameBoard[index] === player);
        });
    }
    
    // Check if the current player has won
    function checkWin() {
        return winCombos.some(combo => {
            if (
                gameBoard[combo[0]] === currentPlayer &&
                gameBoard[combo[1]] === currentPlayer &&
                gameBoard[combo[2]] === currentPlayer
            ) {
                highlightWin(combo);
                return true;
            }
            return false;
        });
    }
    
    // Highlight winning cells
    function highlightWin(combo) {
        combo.forEach(index => {
            cells[index].classList.add('win');
        });
    }
    
    // Check for a draw
    function checkDraw() {
        return gameBoard.every(cell => cell !== '');
    }
    
    // Update score displays
    function updateScores() {
        xScoreDisplay.textContent = scores.X;
        oScoreDisplay.textContent = scores.O;
    }
    
    // Restart the current level
    function restartLevel() {
        scores = { X: 0, O: 0 };
        updateScores();
        initGame();
    }
    
    // Event listeners
    cells.forEach(cell => {
        cell.addEventListener('click', handleCellClick);
    });
    
    resetButton.addEventListener('click', initGame);
    restartButton.addEventListener('click', restartLevel);
    continueButton.addEventListener('click', continueLevelUp);
    
    // Initialize game on load
    initGame();
});