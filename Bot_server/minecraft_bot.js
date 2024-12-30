const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals: { GoalFollow } } = require('mineflayer-pathfinder');
const WebSocket = require('ws');

// Connessione al server Minecraft
const bot = mineflayer.createBot({
    host: '192.168.0.102', // IP del server Minecraft
    port: 52108,           // Porta del server
    username: 'AI_Bot'
});

bot.once('spawn', () => {
    bot.loadPlugin(pathfinder);
    console.log("Bot collegato al server Minecraft");
});

// Inizializza WebSocket server per ricevere comandi
const wss = new WebSocket.Server({ port: 8080 }); // Porta WebSocket
console.log("WebSocket server in ascolto sulla porta 8080");

wss.on('connection', ws => {
    ws.on('message', message => {
        const command = JSON.parse(message);
        console.log("Comando ricevuto:", command);
        switch (command.action) {
            case 'forward':
                bot.setControlState('forward', true);
                setTimeout(() => bot.setControlState('forward', false), command.duration || 1000);
                break;
            case 'jump':
                bot.setControlState('jump', true);
                setTimeout(() => bot.setControlState('jump', false), command.duration || 1000);
                break;
            case 'attack':
                bot.attack(bot.entityAtCursor());
                break;
            // Aggiungi altri comandi se necessario
            default:
                console.log("Comando sconosciuto:", command.action);
        }
    });
});

module.exports = bot;

