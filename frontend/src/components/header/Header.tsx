import { useEffect, useState } from 'react';
import { useGetToken } from '../../apis/queries/auth';
import * as S from './styled';
import WorkflowModal from '../modal/WorkflowModal';
import type { INode } from '../../apis/axios/nodeLib/types';

interface HeaderProps {
  onWorkflowExecute: () => Promise<string[]>;
  selectedNodesCount: number;
  onWorkflowLoad: (workflowData: { name: string; nodes: INode[] }) => void;
  workflowTitle: string;
  onProjectNameEdit?: () => void; // 새로 추가
}

const Header = ({
  onWorkflowExecute,
  selectedNodesCount,
  onWorkflowLoad,
  workflowTitle,
  onProjectNameEdit,
}: HeaderProps) => {
  const { mutateAsync: getTokenSafely } = useGetToken();
  const [isModalOpen, setIsModalOpen] = useState(false);

  useEffect(() => {
    const initializeToken = async () => {
      try {
        // const token = await getTokenSafely();
      } catch (error) {
        console.error();
      }
    };

    initializeToken();
  }, [getTokenSafely]);

  const handleExecute = () => {
    if (selectedNodesCount === 0) {
      alert('실행할 노드를 선택해주세요.');
      return;
    }

    const nodeIds = onWorkflowExecute();
    console.log('선택된 노드 ID 배열:', nodeIds);
  };

  const handleModalOpen = () => {
    setIsModalOpen(true);
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
  };

  const handleWorkflowSelect = (workflowData: { name: string; nodes: INode[] }) => {
    onWorkflowLoad(workflowData);
    setIsModalOpen(false);
  };

  return (
    <>
      <S.HeaderWrapper>
        <S.TitleWrapper>
          <S.Title>{workflowTitle}</S.Title>
          {onProjectNameEdit && (
            <S.EditButton onClick={onProjectNameEdit} title="프로젝트명 변경">
              ✏️
            </S.EditButton>
          )}
        </S.TitleWrapper>

        <S.ButtonWrapper>
          <S.Button variant="blue" onClick={handleModalOpen}>
            워크플로우 불러오기
          </S.Button>
          <S.Button onClick={handleExecute} disabled={selectedNodesCount === 0}>
            워크플로우 실행
          </S.Button>
        </S.ButtonWrapper>
      </S.HeaderWrapper>

      <WorkflowModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        onWorkflowSelect={handleWorkflowSelect}
      />
    </>
  );
};

export default Header;
