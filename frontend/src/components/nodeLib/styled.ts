import styled from 'styled-components';

export const Wrapper = styled.aside`
  width: 250px;
  padding: 16px;
  border-right: 1px solid ${({ theme }) => theme.colors.gray[700]};
  background-color: #17212f;
  flex-shrink: 0;
`;

export const Title = styled.p`
  width: 100%;
  color: #fff;
  ${({ theme }) => theme.typography.size.body.medium.base}
`;

export const NodeWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
  height: calc(100vh - 50px);
  overflow-y: auto;
  margin-top: 24px;
  padding-bottom: 100px;

  scrollbar-width: none;
  -ms-overflow-style: none;
  &::-webkit-scrollbar {
    display: none;
  }
`;

export const NodeList = styled.ul`
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  flex-shrink: 0;
`;
export const NodeTitle = styled.p`
  color: #fff;
  ${({ theme }) => theme.typography.size.body.medium.base}
`;

export const NodeItem = styled.li`
  width: 100%;
  height: auto;
`;

export const ItemButtonWrapper = styled.div`
  position: relative;
  width: 100%;
`;

export const ItemButton = styled.button`
  width: 100%;
  padding: 16px;
  border-radius: 8px;
  background-color: #364151;
  color: #fff;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: #4a5568;
    transform: translateY(-1px);
  }

  &:active {
    background-color: #2d3748;
    transform: translateY(0);
  }
`;

export const Tooltip = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, 0%);
  background-color: #fff;
  color: #000;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  white-space: nowrap;
  z-index: 1000;
  opacity: 0;
  visibility: hidden;
  transition: all 0.2s ease;
  margin-top: 8px;
  width: 200px;
  white-space: normal;
  text-align: center;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);

  /* 화살표 */
  &::before {
    content: '';
    position: absolute;
    top: -5px;
    left: 50%;
    transform: translateX(-50%);
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 5px solid #fff;
  }

  ${ItemButtonWrapper}:hover & {
    opacity: 1;
    visibility: visible;
  }
`;
