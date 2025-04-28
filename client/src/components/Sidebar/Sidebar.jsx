import ChatHistory from './ChatHistory';
import NewChatButton from './NewChatButton';

export default function Sidebar({ chatHistory, onNewChat, onSelectChat, currentChatId }) {
    return (
        <div className="w-64 bg-gray-800 text-white flex flex-col">
            <NewChatButton onClick={onNewChat} />
            <ChatHistory
                chats={chatHistory}
                onSelectChat={onSelectChat}
                currentChatId={currentChatId}
            />
        </div>
    );
}