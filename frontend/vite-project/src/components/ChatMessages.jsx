import { useEffect, useRef } from 'react'
import Message from './Message'
import LoadingIndicator from './LoadingIndicator'

export default function ChatMessages({ messages, isLoading }) {
    const messagesEndRef = useRef(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    return (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
                <Message
                    key={message.id}
                    text={message.text}
                    sender={message.sender}
                />
            ))}

            {isLoading && <LoadingIndicator />}

            <div ref={messagesEndRef} />
        </div>
    )
}