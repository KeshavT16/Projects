
-- =============================================
-- Author:		Keshav Tyagi
-- Create date: Nov 23 2017
-- Description:	Creates a Dimension table for Date
-- =============================================
CREATE PROCEDURE [dbo].[uspCreatePopulateDimdate] 	
AS
BEGIN
BEGIN TRY
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	IF OBJECT_ID('[tmp].[DimDate]','U') IS NOT NULL
	BEGIN
	select 1
	DROP TABLE [tmp].[DimDate]
	END

    CREATE TABLE [tmp].[DimDate](
	[DATEKEY] INT PRIMARY KEY NOT NULL,
	[DayNumberOfWeek] TINYINT NOT NULL,
	[EnglishDayNameOfWeek] VARCHAR(10) NOT NULL,
	[DayNumberOfMonth] TINYINT NOT NULL,
	[EnglishMonthName] VARCHAR(10) NOT NULL,
	[MonthNumberOfYear] TINYINT NOT NULL,
	[CalendarYear] INT NOT NULL
	)

	
	
	IF OBJECT_ID(N'tempdb.dbo.#Date', N'U') IS NOT NULL
	BEGIN
	  DROP TABLE TempDB.dbo.#Date
	END
	SELECT DISTINCT [MONTH],
	[YEAR],
	[DAY],
	[DAY_OF_WEEK],
	CONVERT(INT,SUBSTRING(YEAR,3,2)+right('0'+MONTH,2)+right('0'+DAY,2)) AS  DateKey
	INTO #Date
	FROM [UC].[dbo].[flights]

	INSERT INTO [tmp].[DimDate]
	SELECT DateKey AS [DATEKEY],
	 [DAY_OF_WEEK] AS [DayNumberOfWeek],
	 DATENAME(DW,DATEADD(day,[DAY]-1, convert(date,convert(nvarchar,[datekey]))))AS [EnglishDayNameOfWeek],
	 [Day] AS  [DayNumberOfMonth],
	 DATENAME(month, DATEADD(month, [MONTH]-1, CAST('2008-01-01' AS datetime))) AS [EnglishMonthName],
	 [Month] AS [MonthNumberOfYear],
	 [Year] AS [CalendarYear]
	 FROM #Date

	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM tmp.DimDate)
	 BEGIN
	 IF OBJECT_ID('dbo.DimDate','U') IS NOT NULL
	 BEGIN
	 DROP TABLE dbo.DimDate
	 END
	 ALTER SCHEMA dbo TRANSFER tmp.DimDate
	 END
	 COMMIT TRANSACTION 
	 END TRY

BEGIN CATCH 
  IF (@@TRANCOUNT > 0)
   BEGIN
      ROLLBACK TRANSACTION
      PRINT 'Error detected, all changes reversed'
   END 
    SELECT
        ERROR_NUMBER() AS ErrorNumber,
        ERROR_SEVERITY() AS ErrorSeverity,
        ERROR_STATE() AS ErrorState,
        ERROR_PROCEDURE() AS ErrorProcedure,
        ERROR_LINE() AS ErrorLine,
        ERROR_MESSAGE() AS ErrorMessage
END CATCH

END
