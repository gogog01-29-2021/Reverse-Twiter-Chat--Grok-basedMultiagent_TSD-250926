import styled from 'styled-components';

export const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  width: 100dvw;
  height: 100dvh;
`;

export const ContantsWrapper = styled.div`
  display: flex;
  flex: 1;
  align-items: stretch;
  width: 100%;
  height: calc(100dvh - 65px);
`;

export const MainContent = styled.main`
  flex: 1;
  overflow: auto;
  background-color: #111827;
  display: flex;
  flex-direction: column;
`;

export const ChatWrapper = styled.aside`
  width: 400px;
  border-left: 1px solid ${({ theme }) => theme.colors.gray[700]};
  flex-shrink: 0;
`;

export const SelectedNodesContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 20px;
  width: 100%;
  align-content: flex-start; /* 위쪽부터 정렬 */
  min-height: 0; /* flexbox 문제 해결 */
`;

export const SelectedNodeItem = styled.div`
  background-color: #374151;
  border: 1px solid #4b5563;
  border-radius: 8px;
  padding: 8px 12px;
  min-width: fit-content;
  max-width: 250px;
  height: fit-content;
  transition: all 0.2s ease;

  &:hover {
    background-color: #4b5563;
    border-color: #6b7280;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }
`;

export const NodeContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0; /* overflow 방지 */
`;

export const NodeName = styled.span`
  color: #f3f4f6;
  font-size: 13px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
`;

export const RemoveButton = styled.button`
  background: transparent;
  border: none;
  color: #ef4444;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  padding: 2px;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  flex-shrink: 0;
  transition: all 0.2s ease;

  &:hover {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    transform: scale(1.1);
  }

  &:active {
    transform: scale(0.95);
  }
`;

export const EmptyMessage = styled.div`
  color: #9ca3af;
  font-size: 16px;
  text-align: center;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  font-weight: 500;
`;

// 추가: 노드 개수 표시
export const NodesHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px 0 20px;
  color: #d1d5db;
  font-size: 14px;
  font-weight: 500;
`;

export const NodeCount = styled.span`
  color: #60a5fa;
  background-color: rgba(96, 165, 250, 0.1);
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
`;

// 추가: 전체 삭제 버튼
export const ClearAllButton = styled.button`
  background: transparent;
  border: 1px solid #ef4444;
  color: #ef4444;
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: rgba(239, 68, 68, 0.1);
    color: #f87171;
  }

  &:active {
    transform: scale(0.98);
  }
`;

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

export const WorkflowIdIndicator = styled.span`
  background-color: #e8f0fe;
  color: #1a73e8;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-family: monospace;
`;
