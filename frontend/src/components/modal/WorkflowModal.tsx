import { useEffect, useState } from 'react';
import styled from 'styled-components';
import { instance } from '../../apis/axios';
import type { INode } from '../../apis/axios/nodeLib/types';

interface Workflow {
  id: string;
  name: string;
}

interface WorkflowDetail {
  id: string;
  name: string;
  nodes: INode[];
}

interface WorkflowModalProps {
  isOpen: boolean;
  onClose: () => void;
  onWorkflowSelect?: (workflowData: { name: string; nodes: INode[] }) => void; // 타입 변경
}

const WorkflowModal = ({ isOpen, onClose, onWorkflowSelect }: WorkflowModalProps) => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [loadingWorkflow, setLoadingWorkflow] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchWorkflows();
    }
  }, [isOpen]);

  const fetchWorkflows = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await instance.get('/v1/workflows');

      if (!response) {
        throw new Error('워크플로우를 불러오는데 실패했습니다.');
      }

      setWorkflows(response.data.workflows || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : '알 수 없는 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const fetchWorkflowDetail = async (workflowId: string): Promise<WorkflowDetail> => {
    const response = await instance.get(`/v1/workflows/${workflowId}`);
    return response.data;
  };

  const handleWorkflowClick = (workflow: Workflow) => {
    setSelectedWorkflow(workflow);
  };

  const handleLoadWorkflow = async () => {
    if (!selectedWorkflow || !onWorkflowSelect) return;

    setLoadingWorkflow(true);
    setError(null);

    try {
      const workflowDetail = await fetchWorkflowDetail(selectedWorkflow.id);
      // 워크플로우 이름과 노드 정보 모두 전달
      onWorkflowSelect({
        name: workflowDetail.name,
        nodes: workflowDetail.nodes,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : '워크플로우를 불러오는데 실패했습니다.');
    } finally {
      setLoadingWorkflow(false);
    }
  };

  const handleClose = () => {
    setSelectedWorkflow(null);
    setError(null);
    setLoadingWorkflow(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <ModalBackdrop onClick={handleClose}>
      <WorkflowModalWrapper onClick={(e) => e.stopPropagation()}>
        <ModalHeader>
          <ModalTitle>워크플로우 불러오기</ModalTitle>
          <CloseButton onClick={handleClose}>×</CloseButton>
        </ModalHeader>

        <ModalContent>
          {loading && (
            <LoadingContainer>
              <LoadingSpinner />
              <LoadingText>워크플로우 목록을 불러오는 중...</LoadingText>
            </LoadingContainer>
          )}

          {error && (
            <ErrorContainer>
              <ErrorText>{error}</ErrorText>
              <RetryButton onClick={fetchWorkflows}>다시 시도</RetryButton>
            </ErrorContainer>
          )}

          {!loading && !error && workflows.length === 0 && (
            <EmptyContainer>
              <EmptyText>저장된 워크플로우가 없습니다.</EmptyText>
            </EmptyContainer>
          )}

          {!loading && !error && workflows.length > 0 && (
            <WorkflowList>
              {workflows.map((workflow) => (
                <WorkflowItem
                  key={workflow.id}
                  onClick={() => handleWorkflowClick(workflow)}
                  selected={selectedWorkflow?.id === workflow.id}>
                  <WorkflowInfo>
                    <WorkflowName>{workflow.name}</WorkflowName>
                  </WorkflowInfo>
                  {selectedWorkflow?.id === workflow.id && <SelectedIndicator>✓</SelectedIndicator>}
                </WorkflowItem>
              ))}
            </WorkflowList>
          )}
        </ModalContent>

        <ModalFooter>
          <SecondaryButton onClick={handleClose}>취소</SecondaryButton>
          <PrimaryButton
            onClick={handleLoadWorkflow}
            disabled={!selectedWorkflow || loadingWorkflow}>
            {loadingWorkflow ? '불러오는 중...' : '불러오기'}
          </PrimaryButton>
        </ModalFooter>
      </WorkflowModalWrapper>
    </ModalBackdrop>
  );
};

export default WorkflowModal;

// 기존 스타일 컴포넌트들...
export const ModalBackdrop = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.3s ease-out;

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
`;

export const WorkflowModalWrapper = styled.div`
  position: relative;
  width: 90%;
  max-width: 700px;
  max-height: 90vh;
  padding: 32px;
  border-radius: 20px;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow:
    0 25px 50px -12px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(255, 255, 255, 0.05);
  animation: modalSlideIn 0.3s ease-out;
  overflow-y: auto;

  @keyframes modalSlideIn {
    from {
      opacity: 0;
      transform: scale(0.9) translateY(-20px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 4px;

    &:hover {
      background: rgba(148, 163, 184, 0.5);
    }
  }
`;

export const ModalHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
`;

export const ModalTitle = styled.h2`
  color: #f8fafc;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

export const CloseButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 8px;
  background: rgba(148, 163, 184, 0.1);
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 20px;

  &:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    transform: rotate(90deg);
  }

  &:active {
    transform: rotate(90deg) scale(0.95);
  }
`;

export const ModalContent = styled.div`
  color: #e2e8f0;
  line-height: 1.6;
  min-height: 300px;
`;

export const ModalFooter = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 32px;
  padding-top: 20px;
  border-top: 1px solid rgba(148, 163, 184, 0.2);
`;

export const SecondaryButton = styled.button`
  padding: 10px 20px;
  border-radius: 8px;
  border: 1px solid rgba(148, 163, 184, 0.3);
  background: transparent;
  color: #94a3b8;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;

  &:hover {
    background: rgba(148, 163, 184, 0.1);
    color: #f1f5f9;
    border-color: rgba(148, 163, 184, 0.5);
  }

  &:active {
    transform: translateY(1px);
  }
`;

export const PrimaryButton = styled.button<{ disabled?: boolean }>`
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  background: ${({ disabled }) =>
    disabled ? 'rgba(148, 163, 184, 0.3)' : 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'};
  color: ${({ disabled }) => (disabled ? '#64748b' : '#fff')};
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  transition: all 0.2s ease;
  font-weight: 500;
  box-shadow: ${({ disabled }) => (disabled ? 'none' : '0 4px 14px rgba(59, 130, 246, 0.3)')};

  &:hover {
    ${({ disabled }) =>
      !disabled &&
      `
      background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    `}
  }

  &:active {
    ${({ disabled }) =>
      !disabled &&
      `
      transform: translateY(0);
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    `}
  }
`;

// 새로운 스타일 컴포넌트들
export const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  gap: 16px;
`;

export const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 3px solid rgba(148, 163, 184, 0.3);
  border-top: 3px solid #3b82f6;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

export const LoadingText = styled.p`
  color: #94a3b8;
  margin: 0;
`;

export const ErrorContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  gap: 16px;
`;

export const ErrorText = styled.p`
  color: #ef4444;
  text-align: center;
  margin: 0;
`;

export const RetryButton = styled.button`
  padding: 8px 16px;
  border-radius: 6px;
  border: 1px solid #ef4444;
  background: transparent;
  color: #ef4444;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(239, 68, 68, 0.1);
  }
`;

export const EmptyContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
`;

export const EmptyText = styled.p`
  color: #64748b;
  text-align: center;
  margin: 0;
`;

export const WorkflowList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 400px;
  overflow-y: auto;
`;

export const WorkflowItem = styled.div<{ selected: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-radius: 12px;
  border: 1px solid ${({ selected }) => (selected ? '#3b82f6' : 'rgba(148, 163, 184, 0.2)')};
  background: ${({ selected }) =>
    selected ? 'rgba(59, 130, 246, 0.1)' : 'rgba(148, 163, 184, 0.05)'};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${({ selected }) =>
      selected ? 'rgba(59, 130, 246, 0.15)' : 'rgba(148, 163, 184, 0.1)'};
    border-color: ${({ selected }) => (selected ? '#2563eb' : 'rgba(148, 163, 184, 0.4)')};
  }
`;

export const WorkflowInfo = styled.div`
  flex: 1;
`;

export const WorkflowName = styled.h3`
  color: #f8fafc;
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 4px 0;
`;

export const WorkflowDescription = styled.p`
  color: #94a3b8;
  font-size: 0.9rem;
  margin: 0 0 8px 0;
  line-height: 1.4;
`;

export const WorkflowDate = styled.p`
  color: #64748b;
  font-size: 0.8rem;
  margin: 0;
`;

export const SelectedIndicator = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #3b82f6;
  color: white;
  font-weight: bold;
  font-size: 14px;
`;
