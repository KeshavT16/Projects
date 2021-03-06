
CREATE PROCEDURE [dbo].[usp_CreatePopulateCartLift]
AS

BEGIN
BEGIN TRY

  IF OBJECT_ID('tmp.CartLift','U') IS NOT NULL
  DROP TABLE tmp.CartLift 

 CREATE TABLE [tmp].[CartLift](
	[OBJECTID] int NULL,
	[ADDR_GUID] [varchar](32) NULL,
	[ADDRESSID] [varchar](20) NULL,
	[RFIDTAG] [varchar](25) NULL,
	[CONTAINERSIZE] [varchar](17) NULL,
	[CARTLIFTS] int NULL,
	[RES_VS_COMM] [varchar](5) NULL,
	[YN_LONGTERM_VACANT] [varchar](1) NULL,
	[ADDRESS] [varchar](39) NULL,
	[ZIPCODE] [varchar](5) NULL,
	[NEIGHBORHOOD] [varchar](50) NULL,
	[OESSTUCTTYPE] [varchar](20) NULL,
	[OESCLASSIFICATION] [varchar](20) NULL,
	[AUDLANDUSE] [varchar](100) NULL,
	[ROUTE_DAY] [varchar](50) NULL,
	[ROUTE_type] [varchar] (10) NULL,
	[LATITUDE] float NULL,
	[LONGITUDE] float NULL,
	[DATE] date NULL
) ON [PRIMARY]
  -- creating a refined table by excluding some null values
  INSERT INTO [tmp].[CartLift]
    SELECT CONVERT(INT,[OBJECTID]) AS [OBJECTID]
      ,[ADDR_GUID]
      ,[ADDRESSID]
      ,[RFIDTAG]
      ,[CONTAINERSIZE]
      ,CONVERT(INT,[CARTLIFTS]) AS [CARTLIFTS]
      ,[RES_VS_COMM]
      ,[YN_LONGTERM_VACANT]
      ,[ADDRESS]
      ,[ZIPCODE]
      ,[NEIGHBORHOOD]
      ,[OESSTUCTTYPE]
      ,[OESCLASSIFICATION]
      ,[AUDLANDUSE]
      ,[ROUTE_DAY]
	  ,CASE WHEN [ROUTE_DAY] LIKE '%GREEN%'
	  THEN 'GREEN'
	  WHEN  [ROUTE_DAY] LIKE '%GOLD%'
	  THEN 'GOLD'
	  ELSE 'NONE'
	  END AS [Route_Type]
      ,CONVERT(FLOAT,[LATITUDE]) AS [LATITUDE]
      ,CONVERT(FLOAT,[LONGITUDE]) AS [LONGITUDE]
      ,CONVERT(DATE,[DATE]) AS [DATE]
  FROM [City of Cincinnati].[dbo].[Recycle_Carts_Collection__Tip__Data v5]

    select  REPLACE(convert(varchar,ISNULL([HOUSENUMBER],''))+ISNULL([DIRECTION],'')+ISNULL([STREETNAME],'')+ISNULL([SUFFIX],''),' ','') as ADDRESS
	into #a
    FROM [City of Cincinnati].[dbo].[Mailing InfoGold] 
	UNION
	select  REPLACE(convert(varchar,ISNULL([HOUSENUMBER],''))+ISNULL([DIRECTION],'')+ISNULL([STREETNAME],'')+ISNULL([SUFFIX],''),' ','') as ADDRESS
	FROM [City of Cincinnati].[dbo].[MailingInfoGreen]

	--select top  10 date from tmp.cartlift

	  IF OBJECT_ID('tmp.CartLiftImpact','U') IS NOT NULL
	  DROP TABLE tmp.CartLiftImpact; 


	WITH cte as
	(select distinct address from #a)
	select [CONTAINERSIZE]
      ,[CARTLIFTS]
      ,[RES_VS_COMM]
      ,[YN_LONGTERM_VACANT]
      ,a.[ADDRESS]
      ,[ZIPCODE]
      ,[NEIGHBORHOOD]
      ,[OESSTUCTTYPE]
      ,[OESCLASSIFICATION]
      ,[AUDLANDUSE]
      ,[ROUTE_DAY]
      ,[ROUTE_type]
      ,[LATITUDE]
      ,[LONGITUDE]
      ,[DATE]
	  ,CASE WHEN b.ADDRESS is not null
			THEN 1
			ELSE 0
			END AS 'MailingSent'
			into tmp.CartLiftImpact
	from tmp.CartLift a 
	Left join cte b
	on replace(a.address,' ','') = b.ADDRESS
	WHERE a.date between '2017-06-01' and '2017-12-31'


	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM [tmp].[CartLift])
	 BEGIN
	 IF OBJECT_ID('dbo.CartLift','U') IS NOT NULL
	 BEGIN
	 DROP TABLE [dbo].[CartLift]
	 END
	 ALTER SCHEMA dbo TRANSFER [tmp].[CartLift]
	 END
	 COMMIT TRANSACTION 

	 	 BEGIN TRANSACTION 
	 IF EXISTS( Select 1 FROM [tmp].CartLiftImpact)
	 BEGIN
	 IF OBJECT_ID('dbo.CartLiftImpact','U') IS NOT NULL
	 BEGIN
	 DROP TABLE [dbo].CartLiftImpact
	 END
	 ALTER SCHEMA dbo TRANSFER [tmp].CartLiftImpact
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