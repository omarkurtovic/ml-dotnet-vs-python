using System;
using System.Collections.Generic;
using System.Text;

namespace SharedCL.Shared.Models
{
    public class Result
    {
        public bool IsSuccess { get; set; }
        public string? Message { get; set; }
        public FailureReason Reason { get; set; }


        public static Result Success()
        {
            return new Result { IsSuccess = true };
        }
        public static Result Success(string message)
        {
            return new Result { IsSuccess = true, Message = message };
        }

        public static Result Failure(string message, FailureReason reason = FailureReason.UnknownError)
        {
            return new Result
            {
                IsSuccess = false,
                Message = message,
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
        public static Result<T> Success(T data, string message)
        {
            return new Result<T> { IsSuccess = true, Data = data, Message = message };
        }

        public new static Result<T> Failure(string message, FailureReason reason = FailureReason.UnknownError)
        {
            return new Result<T>
            {
                IsSuccess = false,
                Message = message,
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

