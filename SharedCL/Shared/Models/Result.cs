using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.Shared.Models
{
    public class Result
    {
        public bool IsSuccess { get; set; }
        public string? ErrorMessage { get; set; }
        public FailureReason Reason { get; set; }


        public static Result Success()
        {
            return new Result { IsSuccess = true };
        }

        public static Result Failure(string errorMessage, FailureReason reason = FailureReason.UnknownError)
        {
            return new Result
            {
                IsSuccess = false,
                ErrorMessage = errorMessage,
                Reason = reason
            };
        }
    }

    public class Result<T> : Result
    {
        public T? Data { get; set; }

        public static Result<T> Success(T data)
        {
            return new Result<T> { IsSuccess = true, Data = data };
        }

        public new static Result<T> Failure(string errorMessage, FailureReason reason = FailureReason.UnknownError)
        {
            return new Result<T>
            {
                IsSuccess = false,
                ErrorMessage = errorMessage,
                Reason = reason
            };
        }
    }
    public enum FailureReason
    {
        NotFound,
        ValidationError,
        Unauthorized,
        UnknownError
    }
}

