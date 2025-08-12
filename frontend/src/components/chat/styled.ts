import styled from 'styled-components';

export const ChatContainer = styled.div`
  width: 100%;
  max-width: 400px;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background-color: #18212f;

  iframe {
    width: 100%;
    height: 100%;
  }
`;

export const Messages = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  /* background: #f9f9f9; */
`;

export const Message = styled.div<{ isUser?: boolean }>`
  max-width: 70%;
  margin: 6px 0;
  padding: 10px 14px;
  border-radius: 8px;
  color: #fff;
  align-self: ${(props) => (props.isUser ? 'flex-end' : 'flex-start')};
  background: ${(props) => (props.isUser ? '#4c5563' : '#384151')};
`;

export const InputWrapper = styled.div`
  display: flex;
  padding: 12px;
  border-top: 1px solid #ccc;
  background: white;
`;

export const Input = styled.input`
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 20px;
  background: #f0f0f0;
  font-size: 16px;
`;

export const SendButton = styled.button`
  margin-left: 8px;
  padding: 0 16px;
  border: none;
  border-radius: 20px;
  background-color: #b49884;
  color: white;
  cursor: pointer;
`;
// Chat/styled.ts
export const WorkflowIndicator = styled.div`
  padding: 8px 16px;
  background-color: #f0f8ff;
  border-bottom: 1px solid #e1e8ed;
  font-size: 12px;
`;

export const WorkflowStatus = styled.div`
  color: #1a73e8;
  font-weight: 500;
`;

export const WorkflowMessage = styled.div`
  background-color: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border-left: 4px solid #1a73e8;
`;

export const WorkflowHeader = styled.div`
  font-weight: 600;
  color: #1a73e8;
  margin-bottom: 8px;
  font-size: 14px;
`;

export const WorkflowContent = styled.div`
  color: #333;
  line-height: 1.5;
  margin-bottom: 8px;
`;

export const WorkflowId = styled.div`
  font-size: 11px;
  color: #666;
  font-family: monospace;
  background-color: #f1f3f4;
  padding: 4px 8px;
  border-radius: 4px;
  display: inline-block;
`;
