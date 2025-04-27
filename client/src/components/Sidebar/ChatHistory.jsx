export default function ChatHistory({ chats, onSelectChat, currentChatId }) {
    return (
        <div className="flex-1 overflow-y-auto">
            <h2 className="px-4 py-2 text-sm font-semibold text-gray-400">Previous Chats</h2>
            <ul>
                {chats.map(chat => (
                    <li
                        key={chat.id}
                        className={`px-4 py-2 cursor-pointer hover:bg-gray-700 ${currentChatId === chat.id ? 'bg-gray-700' : ''}`}
                        onClick={() => onSelectChat(chat)}
                    >
                        <div className="font-medium">{chat.title}</div>
                        <div className="text-xs text-gray-400">{chat.date}</div>
                    </li>
                ))}
            </ul>
        </div>
    );
}