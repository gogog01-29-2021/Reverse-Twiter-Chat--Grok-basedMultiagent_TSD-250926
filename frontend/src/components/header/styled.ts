import styled from 'styled-components';

export const HeaderWrapper = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  width: 100%;
  height: 65px;
  padding: 0px 16px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.gray[700]};
  background-color: #1f2937;
`;

export const Title = styled.p`
  color: #fff;
  padding: 0px 16px;
  ${({ theme }) => theme.typography.size.headline.h3};
`;

export const ButtonWrapper = styled.div`
  display: flex;
  flex-direction: row;
  gap: 16px;
`;

export const Button = styled.button<{ disabled?: boolean; variant?: 'green' | 'blue' }>`
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  color: #fff;
  background-color: ${({ disabled, variant = 'green' }) => {
    if (disabled) return '#9ca3af';
    return variant === 'blue' ? '#0068fe' : '#16a34a';
  }};
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  transition: all 0.2s ease;
  ${({ theme }) => theme.typography.size.button.base};

  &:hover {
    background-color: ${({ disabled, variant = 'green' }) => {
      if (disabled) return '#9ca3af';
      return variant === 'blue' ? '#0052cc' : '#15803d';
    }};
    transform: ${({ disabled }) => (disabled ? 'none' : 'translateY(-1px)')};
  }

  &:active {
    background-color: ${({ disabled, variant = 'green' }) => {
      if (disabled) return '#9ca3af';
      return variant === 'blue' ? '#003d99' : '#14532d';
    }};
    transform: ${({ disabled }) => (disabled ? 'none' : 'translateY(0)')};
  }

  &:focus {
    outline: none;
    box-shadow: ${({ disabled, variant = 'green' }) => {
      if (disabled) return 'none';
      return variant === 'blue'
        ? '0 0 0 3px rgba(0, 104, 254, 0.3)'
        : '0 0 0 3px rgba(22, 163, 74, 0.3)';
    }};
  }
`;

export const WorkflowModal = styled.dialog`
  position: absolute;
  top: 50%;
  left: 50%;
  translate: -50% -50%;
  width: 500px;
  height: 500px;
  padding: 16px;
  border-radius: 16px;
  background-color: #17212f;
  border: 1px solid ${({ theme }) => theme.colors.gray[700]};

  box-shadow:
    0 25px 50px -12px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(0, 0, 0, 0.05);
`;

export const TitleWrapper = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

export const EditButton = styled.button`
  background: transparent;
  border: 1px solid rgba(148, 163, 184, 0.3);
  border-radius: 8px;
  padding: 6px 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 14px;

  &:hover {
    background: rgba(148, 163, 184, 0.1);
    border-color: rgba(148, 163, 184, 0.5);
    transform: scale(1.05);
  }

  &:active {
    transform: scale(0.98);
  }
`;
