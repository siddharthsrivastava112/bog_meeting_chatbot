export default function ChatHeader({ title = 'DeepSeek Chat' }) {
    return (
        <header className="bg-white border-b border-gray-200 py-3 px-4">
            <h1 className="text-lg font-semibold text-gray-800">{title}</h1>
        </header>
    )
}