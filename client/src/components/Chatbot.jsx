import { useState } from 'react';
import ChatHeader from './ChatHeader';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';

export default function ChatBot() {
    const [messages, setMessages] = useState([
        { id: 1, text: 'Hello! How can I help you today?', sender: 'bot' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const handleSendMessage = (messageText) => {
        if (messageText.trim() === '') return;

        // Add user message
        const newUserMessage = {
            id: messages.length + 1,
            text: messageText,
            sender: 'user'
        };
        setMessages([...messages, newUserMessage]);
        setIsLoading(true);

        // Simulate bot response
        setTimeout(() => {
            const botResponses = [
                "I understand your question about '" + messageText + "'. Let me think...",
                "That's an interesting point about '" + messageText + "'. Here's what I know...",
                "I've analyzed your query regarding '" + messageText + "'. My response is..."
            ];
            const randomResponse = botResponses[Math.floor(Math.random() * botResponses.length)];

            const newBotMessage = {
                id: messages.length + 2,
                text: randomResponse,
                sender: 'bot'
            };
            setMessages(prev => [...prev, newBotMessage]);
            setIsLoading(false);
        }, 1500);
    };

    return (
        <div className="flex flex-col h-full bg-gray-50">
            <ChatHeader title="Current Chat" />
            <ChatMessages messages={messages} isLoading={isLoading} />
            <ChatInput onSendMessage={handleSendMessage} />
        </div>
    );
}