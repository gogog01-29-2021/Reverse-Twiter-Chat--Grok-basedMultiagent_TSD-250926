import type { INode } from '../../apis/axios/nodeLib/types';
import { useGetNodes } from '../../apis/queries/nodeLib';
import * as S from './styled';

interface NodeLibProps {
  onNodeSelect: (node: INode) => void;
}

const NodeLib = ({ onNodeSelect }: NodeLibProps) => {
  const { data } = useGetNodes();

  return (
    <S.Wrapper>
      <S.Title>노드 라이브러리</S.Title>
      <S.NodeWrapper>
        {data?.groups?.map((e, index) => (
          <S.NodeList key={index}>
            <S.NodeTitle>{e.name}</S.NodeTitle>
            {e?.nodes?.map((i, nodeIndex) => (
              <S.NodeItem key={nodeIndex}>
                <S.ItemButtonWrapper>
                  <S.ItemButton onClick={() => onNodeSelect(i)}>{i.name}</S.ItemButton>
                  <S.Tooltip>{i.description}</S.Tooltip>
                </S.ItemButtonWrapper>
              </S.NodeItem>
            ))}
          </S.NodeList>
        ))}
      </S.NodeWrapper>
    </S.Wrapper>
  );
};

export default NodeLib;
