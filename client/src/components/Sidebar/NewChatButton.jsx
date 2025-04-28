export default function NewChatButton({ onClick }) {
    return (
        <button
            onClick={onClick}
            className="m-2 p-2 flex items-center gap-2 rounded-lg border border-gray-600 hover:bg-gray-700"
        >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            New Chat
        </button>
    );
}