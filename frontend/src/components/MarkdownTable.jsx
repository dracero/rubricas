import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

/**
 * A robust component that uses react-markdown and remark-gfm
 * to render Markdown content, including tables, beautifully with Tailwind.
 */
const MarkdownTable = ({ content }) => {
    if (!content) return null;

    return (
        <div className="markdown-content">
            <ReactMarkdown 
                remarkPlugins={[remarkGfm]}
                components={{
                    // Style tables with Tailwind
                    table: ({ node, ...props }) => (
                        <div className="overflow-x-auto my-4 rounded-lg border border-gray-200">
                            <table className="min-w-full divide-y divide-gray-200 text-sm" {...props} />
                        </div>
                    ),
                    thead: ({ node, ...props }) => <thead className="bg-gray-50 text-gray-700 font-bold uppercase tracking-wider" {...props} />,
                    tbody: ({ node, ...props }) => <tbody className="bg-white divide-y divide-gray-100" {...props} />,
                    tr: ({ node, ...props }) => <tr className="even:bg-gray-50/50" {...props} />,
                    th: ({ node, ...props }) => <th className="px-4 py-3 text-left border-b border-gray-200" {...props} />,
                    td: ({ node, ...props }) => <td className="px-4 py-3 text-gray-800 break-words max-w-xs align-top" {...props} />,
                    // Style other common elements
                    p: ({ node, ...props }) => <p className="mb-4 last:mb-0 leading-relaxed text-gray-800" {...props} />,
                    ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-4 space-y-1" {...props} />,
                    ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-4 space-y-1" {...props} />,
                    li: ({ node, ...props }) => <li className="text-gray-800" {...props} />,
                    h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mb-4 mt-6 text-gray-900 border-b pb-2" {...props} />,
                    h2: ({ node, ...props }) => <h2 className="text-xl font-semibold mb-3 mt-5 text-gray-900" {...props} />,
                    h3: ({ node, ...props }) => <h3 className="text-lg font-medium mb-2 mt-4 text-gray-900" {...props} />,
                    strong: ({ node, ...props }) => <strong className="font-bold text-gray-900" {...props} />,
                    em: ({ node, ...props }) => <em className="italic" {...props} />,
                    code: ({ node, inline, ...props }) => 
                        inline 
                            ? <code className="bg-gray-100 px-1 rounded text-pink-600 font-mono text-xs" {...props} />
                            : <code className="block bg-gray-900 text-gray-100 p-3 rounded-lg font-mono text-xs my-4 overflow-x-auto" {...props} />,
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
};

export default MarkdownTable;
