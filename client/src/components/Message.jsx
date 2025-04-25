export default function Message({ text, sender }) {
    return (
        <div className={`flex ${sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
                className={`max-w-3/4 rounded-lg px-4 py-2 ${sender === 'user'
                        ? 'bg-blue-500 text-white rounded-br-none'
                        : 'bg-gray-200 text-gray-800 rounded-bl-none'
                    }`}
            >
                {text}
            </div>
        </div>
    )
}